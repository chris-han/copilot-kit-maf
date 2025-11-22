import { Mail as LucideMail, LucideProps } from 'lucide-react';

const MailLine = ({ className, ...props }: LucideProps) => {
  return <LucideMail className={className} {...props} />;
};

export default MailLine;