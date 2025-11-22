import { MoreVertical as LucideMoreVertical, LucideProps } from 'lucide-react';

const MoreDotIcon = ({ className, ...props }: LucideProps) => {
  return <LucideMoreVertical className={className} {...props} />;
};

export default MoreDotIcon;