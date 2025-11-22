import { MoreVertical as LucideMoreVertical, LucideProps } from 'lucide-react';

const MoreVertical = ({ className, ...props }: LucideProps) => {
  return <LucideMoreVertical className={className} {...props} />;
};

export default MoreVertical;